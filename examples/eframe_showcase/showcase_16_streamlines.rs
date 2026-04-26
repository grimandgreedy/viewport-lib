//! Showcase 16: Streamlines & Tubes
//!
//! Demonstrates `PolylineItem` (thin streamlines) and `StreamtubeItem` (3-D tube
//! rendering) for flow visualization. Synthetic streamlines are seeded from a grid
//! of starting points and integrated through a simple analytic vortex/dipole
//! velocity field. The same path data is submitted either as `PolylineItem` or
//! `StreamtubeItem` depending on a toggle.

use crate::App;
use eframe::egui;
use viewport_lib::{BuiltinColormap, ColormapId, PolylineItem, StreamtubeItem};

impl App {
    /// One-time CPU setup for Showcase 16.
    ///
    /// Generates stream paths from the current seed count and integration step.
    /// The result is stored in `stream_paths` and `stream_scalars` (per-vertex speed).
    pub(crate) fn build_stream_scene(&mut self) {
        let (paths, scalars) = integrate_streamlines(self.stream_seed_count, self.stream_step_size);
        self.stream_paths = paths;
        self.stream_scalars = scalars;
        self.stream_built = true;
    }

    // -------------------------------------------------------------------------
    // Controls panel
    // -------------------------------------------------------------------------

    pub(crate) fn controls_streamlines(&mut self, ui: &mut egui::Ui) {
        ui.label("Render mode:");
        ui.horizontal(|ui| {
            if ui.radio(!self.stream_use_tubes, "Polylines").clicked() {
                self.stream_use_tubes = false;
            }
            if ui.radio(self.stream_use_tubes, "Tubes").clicked() {
                self.stream_use_tubes = true;
            }
        });

        ui.separator();

        if self.stream_use_tubes {
            ui.label("Tube radius:");
            ui.add(egui::Slider::new(&mut self.stream_tube_radius, 0.01..=0.3).step_by(0.01));
        } else {
            ui.label("Line width (px):");
            ui.add(egui::Slider::new(&mut self.stream_line_width, 0.5..=8.0).step_by(0.5));
        }

        ui.separator();
        ui.label("Coloring:");
        ui.horizontal(|ui| {
            if ui
                .radio(!self.stream_color_by_speed, "Flat color")
                .clicked()
            {
                self.stream_color_by_speed = false;
            }
            if ui.radio(self.stream_color_by_speed, "Speed").clicked() {
                self.stream_color_by_speed = true;
            }
        });

        if self.stream_color_by_speed {
            ui.label("Colormap:");
            for (preset, label) in [
                (BuiltinColormap::Viridis, "Viridis"),
                (BuiltinColormap::Plasma, "Plasma"),
                (BuiltinColormap::Coolwarm, "Coolwarm"),
                (BuiltinColormap::Rainbow, "Rainbow"),
                (BuiltinColormap::Greyscale, "Greyscale"),
            ] {
                if ui.radio(self.stream_colormap == preset, label).clicked() {
                    self.stream_colormap = preset;
                }
            }
        } else {
            ui.label("Line color:");
            let mut rgb = [
                self.stream_flat_color[0],
                self.stream_flat_color[1],
                self.stream_flat_color[2],
            ];
            if ui.color_edit_button_rgb(&mut rgb).changed() {
                self.stream_flat_color[0] = rgb[0];
                self.stream_flat_color[1] = rgb[1];
                self.stream_flat_color[2] = rgb[2];
            }
        }

        ui.separator();
        ui.label("Seed count:");
        let seed_response =
            ui.add(egui::Slider::new(&mut self.stream_seed_count, 8..=64).step_by(4.0));
        ui.label("Integration step:");
        let step_response =
            ui.add(egui::Slider::new(&mut self.stream_step_size, 0.02..=0.2).step_by(0.01));

        // Regenerate paths when seed or step sliders are released.
        if seed_response.drag_stopped() || step_response.drag_stopped() {
            self.build_stream_scene();
        }
    }

    // -------------------------------------------------------------------------
    // Frame-data helpers (called from build_frame_data)
    // -------------------------------------------------------------------------

    /// Build a `PolylineItem` from cached stream paths and current control state.
    pub(crate) fn make_stream_polyline_item(&self) -> PolylineItem {
        let (positions, strip_lengths, scalars) =
            flatten_paths(&self.stream_paths, &self.stream_scalars);
        let mut item = PolylineItem::default();
        item.positions = positions;
        item.strip_lengths = strip_lengths;
        item.line_width = self.stream_line_width;
        if self.stream_color_by_speed {
            item.scalars = scalars;
            item.colormap_id = Some(ColormapId(self.stream_colormap as usize));
        } else {
            item.default_color = self.stream_flat_color;
        }
        item
    }

    /// Build a `StreamtubeItem` from cached stream paths and current control state.
    pub(crate) fn make_stream_tube_item(&self) -> StreamtubeItem {
        let (positions, strip_lengths, _scalars) =
            flatten_paths(&self.stream_paths, &self.stream_scalars);
        let mut item = StreamtubeItem::default();
        item.positions = positions;
        item.strip_lengths = strip_lengths;
        item.radius = self.stream_tube_radius;
        item.color = self.stream_flat_color;
        item
    }
}

// ---------------------------------------------------------------------------
// Path generation
// ---------------------------------------------------------------------------

/// Integrate `seed_count` streamlines through an analytic vortex field.
///
/// Seeds are placed on a ring in the XY plane. Integration uses a simple
/// fixed-step Euler method for clarity; each streamline runs for up to 200
/// steps or until it exits a bounding sphere.
///
/// Returns `(paths, scalars)` where each `paths[i]` is the list of 3-D
/// positions for streamline `i`, and `scalars[i]` holds the per-vertex speed
/// (magnitude of the velocity vector at that point).
fn integrate_streamlines(seed_count: usize, step_size: f32) -> (Vec<Vec<[f32; 3]>>, Vec<Vec<f32>>) {
    use std::f32::consts::TAU;

    let mut paths: Vec<Vec<[f32; 3]>> = Vec::with_capacity(seed_count);
    let mut scalars: Vec<Vec<f32>> = Vec::with_capacity(seed_count);

    let seed_radius = 2.5_f32;
    let max_steps = 200_usize;
    let bound = 6.0_f32;

    for i in 0..seed_count {
        let angle = (i as f32 / seed_count as f32) * TAU;
        let sx = seed_radius * angle.cos();
        let sy = seed_radius * angle.sin();
        let sz = (i as f32 / seed_count as f32 - 0.5) * 2.0; // spread in Z

        let mut p = [sx, sy, sz];
        let mut path: Vec<[f32; 3]> = Vec::with_capacity(max_steps);
        let mut spd: Vec<f32> = Vec::with_capacity(max_steps);

        path.push(p);
        spd.push(velocity_magnitude(p));

        for _ in 0..max_steps {
            let v = velocity(p);
            let mag = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            if mag < 1e-4 {
                break;
            }
            p[0] += v[0] * step_size;
            p[1] += v[1] * step_size;
            p[2] += v[2] * step_size;

            let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
            if r > bound {
                break;
            }
            path.push(p);
            spd.push(velocity_magnitude(p));
        }

        if path.len() >= 2 {
            paths.push(path);
            scalars.push(spd);
        }
    }

    (paths, scalars)
}

/// Analytic velocity field: a spiraling vortex with a gentle upward drift.
///
/// The field is:
///   vx = -y / (r² + ε)  (tangential)
///   vy =  x / (r² + ε)
///   vz =  0.3 * exp(-r² / 4)  (upwelling near origin)
fn velocity(p: [f32; 3]) -> [f32; 3] {
    let x = p[0];
    let y = p[1];
    let z = p[2];
    let r2 = x * x + y * y + z * z;
    let eps = 0.5_f32;
    let denom = r2 + eps;
    let vx = -y / denom;
    let vy = x / denom;
    let vz = 0.3 * (-r2 / 4.0).exp();
    [vx, vy, vz]
}

fn velocity_magnitude(p: [f32; 3]) -> f32 {
    let v = velocity(p);
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

/// Flatten a list of per-streamline paths into concatenated buffers.
///
/// Returns `(positions, strip_lengths, scalars)` suitable for direct use in
/// `PolylineItem` and `StreamtubeItem`.
fn flatten_paths(
    paths: &[Vec<[f32; 3]>],
    scalars: &[Vec<f32>],
) -> (Vec<[f32; 3]>, Vec<u32>, Vec<f32>) {
    let total: usize = paths.iter().map(|p| p.len()).sum();
    let mut positions = Vec::with_capacity(total);
    let mut strip_lengths = Vec::with_capacity(paths.len());
    let mut flat_scalars = Vec::with_capacity(total);

    for (path, spd) in paths.iter().zip(scalars.iter()) {
        positions.extend_from_slice(path);
        strip_lengths.push(path.len() as u32);
        flat_scalars.extend_from_slice(spd);
    }

    (positions, strip_lengths, flat_scalars)
}
