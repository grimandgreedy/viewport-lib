//! Showcase 19: Matcap Shading.
//!
//! Displays all eight built-in matcap presets (four blendable, four static) plus a
//! custom matcap generated procedurally at build time.  The controls panel lets you
//! change the `base_color` used by the blendable presets and regenerate the custom
//! matcap with a different hue.

use crate::App;
use crate::geometry::make_uv_sphere;
use eframe::egui;
use viewport_lib::{BuiltinMatcap, Material, MeshId, ViewportRenderer, scene::Scene};

/// All eight built-in presets with their display name and blendable flag.
pub(crate) const BUILTIN_PRESETS: [(BuiltinMatcap, &str, bool); 8] = [
    (BuiltinMatcap::Clay,    "Clay",    true),
    (BuiltinMatcap::Wax,     "Wax",     true),
    (BuiltinMatcap::Candy,   "Candy",   true),
    (BuiltinMatcap::Flat,    "Flat",    true),
    (BuiltinMatcap::Ceramic, "Ceramic", false),
    (BuiltinMatcap::Jade,    "Jade",    false),
    (BuiltinMatcap::Mud,     "Mud",     false),
    (BuiltinMatcap::Normal,  "Normal",  false),
];

impl App {
    /// Build Showcase 19: Matcap Shading demo.
    ///
    /// Layout:
    ///   Top row    (z = +1.5): Clay · Wax · Candy · Flat           — blendable
    ///   Bottom row (z = -1.5): Ceramic · Jade · Mud · Normal       — static
    ///   Center-front (y = -3.5): Custom procedural matcap upload demo
    pub(crate) fn build_matcap_scene(&mut self, renderer: &mut ViewportRenderer) {
        self.matcap_scene = Scene::new();

        // The non-instanced path stores per-object GPU state (object_uniform_buf,
        // object_bind_group) directly on the GpuMesh.  All objects sharing the same
        // mesh_index would therefore overwrite each other during prepare().
        // Each sphere needs its own GpuMesh so it gets independent GPU state.
        let sphere = make_uv_sphere(48, 24, 1.0);
        let upload_sphere = |renderer: &mut ViewportRenderer, device: &eframe::wgpu::Device| {
            MeshId::from_index(
                renderer
                    .resources_mut()
                    .upload_mesh_data(device, &sphere)
                    .expect("matcap sphere mesh upload"),
            )
        };

        // Initialise built-ins before calling builtin_matcap_id.
        renderer
            .resources_mut()
            .ensure_matcaps_initialized(&self.device, &self.queue);

        let x_positions: [f32; 4] = [-4.5, -1.5, 1.5, 4.5];

        for (i, (preset, label, _blendable)) in BUILTIN_PRESETS.iter().enumerate() {
            let col = i % 4;
            let row = i / 4;
            let x = x_positions[col];
            let z = if row == 0 { 1.5_f32 } else { -1.5_f32 };

            let sphere_id = upload_sphere(renderer, &self.device);
            let matcap_id = renderer.resources().builtin_matcap_id(*preset);
            let mat = Material {
                base_color: self.matcap_blendable_color,
                matcap_id: Some(matcap_id),
                ..Material::default()
            };
            let node_id = self.matcap_scene.add_named(
                *label,
                Some(sphere_id),
                glam::Mat4::from_translation(glam::Vec3::new(x, 0.0, z)),
                mat,
            );
            self.matcap_builtin_node_ids[i] = node_id;
        }

        // Custom matcap: procedurally generated at startup with the current hue.
        let custom_rgba = generate_custom_matcap(256, self.matcap_custom_hue);
        let custom_id = renderer
            .resources_mut()
            .upload_matcap(&self.device, &self.queue, &custom_rgba, false)
            .expect("custom matcap upload");
        self.matcap_custom_id = Some(custom_id);

        let custom_sphere_id = upload_sphere(renderer, &self.device);
        let custom_node = self.matcap_scene.add_named(
            "Custom",
            Some(custom_sphere_id),
            glam::Mat4::from_translation(glam::Vec3::new(0.0, -3.5, 0.0)),
            Material {
                matcap_id: Some(custom_id),
                ..Material::default()
            },
        );
        self.matcap_custom_node = Some(custom_node);

        self.matcap_built = true;
    }

    /// Re-upload the custom matcap with the current hue and update the scene node.
    pub(crate) fn rebuild_custom_matcap(&mut self, renderer: &mut ViewportRenderer) {
        let rgba = generate_custom_matcap(256, self.matcap_custom_hue);
        let id = renderer
            .resources_mut()
            .upload_matcap(&self.device, &self.queue, &rgba, false)
            .expect("custom matcap re-upload");
        self.matcap_custom_id = Some(id);
        if let Some(node_id) = self.matcap_custom_node {
            self.matcap_scene.set_material(
                node_id,
                Material {
                    matcap_id: Some(id),
                    ..Material::default()
                },
            );
        }
    }

    /// Push the current `matcap_blendable_color` to all blendable preset nodes.
    pub(crate) fn update_matcap_blendable_colors(&mut self, renderer: &mut ViewportRenderer) {
        for (i, (preset, _, blendable)) in BUILTIN_PRESETS.iter().enumerate() {
            if !blendable {
                continue;
            }
            let matcap_id = renderer.resources().builtin_matcap_id(*preset);
            self.matcap_scene.set_material(
                self.matcap_builtin_node_ids[i],
                Material {
                    base_color: self.matcap_blendable_color,
                    matcap_id: Some(matcap_id),
                    ..Material::default()
                },
            );
        }
    }

    pub(crate) fn controls_matcap(&mut self, ui: &mut egui::Ui, frame: &eframe::Frame) {
        ui.label("Eight built-in matcap presets arranged in two rows:");
        ui.label("  North (z+): Clay · Wax · Candy · Flat  (blendable)");
        ui.label("  South (z-): Ceramic · Jade · Mud · Normal  (static)");
        ui.label("  Front (y-): custom procedural upload");

        ui.separator();
        ui.label("Blendable base color:");
        ui.horizontal(|ui| {
            let mut col = self.matcap_blendable_color;
            let changed = ui
                .color_edit_button_rgb(&mut col)
                .changed();
            if changed {
                self.matcap_blendable_color = col;
                let rs = frame.wgpu_render_state().expect("wgpu must be enabled");
                let mut guard = rs.renderer.write();
                let renderer = guard
                    .callback_resources
                    .get_mut::<ViewportRenderer>()
                    .expect("ViewportRenderer");
                self.update_matcap_blendable_colors(renderer);
            }
        });
        ui.label("(tints Clay, Wax, Candy, Flat)");

        ui.separator();
        ui.label("Custom matcap hue:");
        let hue_changed = ui
            .add(egui::Slider::new(&mut self.matcap_custom_hue, 0.0..=360.0)
                .suffix("°")
                .step_by(1.0))
            .changed();
        if ui.button("Rebuild custom matcap").clicked() || hue_changed {
            let rs = frame.wgpu_render_state().expect("wgpu must be enabled");
            let mut guard = rs.renderer.write();
            let renderer = guard
                .callback_resources
                .get_mut::<ViewportRenderer>()
                .expect("ViewportRenderer");
            self.rebuild_custom_matcap(renderer);
        }
        ui.label("Generated via upload_matcap().");
    }
}

// ---------------------------------------------------------------------------
// Custom matcap generator
// ---------------------------------------------------------------------------

/// Generate a `size×size` RGBA matcap image with a metallic look at `hue_deg` (0..360).
///
/// Pixels outside the unit circle are transparent so the object silhouette is preserved.
pub(crate) fn generate_custom_matcap(size: usize, hue_deg: f32) -> Vec<u8> {
    let mut out = Vec::with_capacity(size * size * 4);
    let hue = hue_deg / 360.0;
    let s = (size - 1) as f32;

    for row in 0..size {
        for col in 0..size {
            // row 0 = top of matcap = ny = +1
            let nx = (col as f32 / s) * 2.0 - 1.0;
            let ny = 1.0 - (row as f32 / s) * 2.0;
            let r2 = nx * nx + ny * ny;

            if r2 > 1.0 {
                out.extend_from_slice(&[0, 0, 0, 0]);
                continue;
            }

            let nz = (1.0 - r2).sqrt();

            // Single directional light from upper-left
            let (lx, ly, lz) = normalise(-0.5_f32, 0.7, 0.5);
            let diffuse = (nx * lx + ny * ly + nz * lz).max(0.0);

            // Specular: half-vector with view direction (0,0,1)
            let (hx, hy, hz) = normalise(lx, ly, lz + 1.0);
            let spec = (nx * hx + ny * hy + nz * hz).max(0.0).powf(80.0);

            let (r, g, b) = hsv_to_rgb(hue, 0.7, 0.15 + 0.6 * diffuse);
            let r = ((r + spec) * 255.0).clamp(0.0, 255.0) as u8;
            let g = ((g + spec * 0.9) * 255.0).clamp(0.0, 255.0) as u8;
            let b = ((b + spec) * 255.0).clamp(0.0, 255.0) as u8;
            out.extend_from_slice(&[r, g, b, 255]);
        }
    }
    out
}

fn normalise(x: f32, y: f32, z: f32) -> (f32, f32, f32) {
    let len = (x * x + y * y + z * z).sqrt();
    (x / len, y / len, z / len)
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (f32, f32, f32) {
    let h6 = h * 6.0;
    let i = h6.floor() as u32 % 6;
    let f = h6 - h6.floor();
    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));
    match i {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    }
}
