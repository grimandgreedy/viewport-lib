//! Showcase 28: Curve Network Quantities
//!
//! Demonstrates the Phase 11 curve-network quantity system on a single helix
//! polyline. Each mode shows one of the new attributes added to `PolylineItem`:
//!
//! - **EdgeScalar**: per-edge scalar -> colormap (flat constant color per segment)
//! - **NodeColor**: per-node direct RGBA (smooth gradient along strip)
//! - **EdgeColor**: per-edge direct RGBA (flat constant color per segment)
//! - **NodeRadius**: per-node line width that varies along the strip
//! - **NodeVectors**: tangent arrows at each node (auto-rendered via GlyphItem)
//! - **EdgeVectors**: normal arrows at each segment midpoint

use crate::App;
use eframe::egui;
use std::f32::consts::TAU;
use viewport_lib::{ColormapId, BuiltinColormap, PolylineItem};

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum CnqMode {
    EdgeScalar,
    NodeColor,
    EdgeColor,
    NodeRadius,
    NodeVectors,
    EdgeVectors,
}

impl App {
    pub(crate) fn controls_cnq(&mut self, ui: &mut egui::Ui) {
        ui.label("Quantity mode:");
        for (mode, label) in [
            (CnqMode::EdgeScalar,  "Edge scalar (LUT coloring)"),
            (CnqMode::NodeColor,   "Node color (direct RGBA)"),
            (CnqMode::EdgeColor,   "Edge color (direct RGBA)"),
            (CnqMode::NodeRadius,  "Node radius (varying width)"),
            (CnqMode::NodeVectors, "Node vectors (tangent arrows)"),
            (CnqMode::EdgeVectors, "Edge vectors (normal arrows)"),
        ] {
            if ui.radio(self.cnq_mode == mode, label).clicked() {
                self.cnq_mode = mode;
            }
        }

        ui.separator();
        ui.label("Base line width:");
        ui.add(egui::Slider::new(&mut self.cnq_line_width, 1.0..=10.0).step_by(0.5));
    }

    /// Build a `PolylineItem` demonstrating the currently selected quantity mode.
    ///
    /// Uses a helix with 120 nodes as the base geometry.  All quantity data is
    /// derived analytically from the helix parameter `t` in [0, 2π].
    pub(crate) fn make_cnq_polyline_item(&self) -> PolylineItem {
        let n = 120usize;
        let turns = 3.0_f32;
        let radius = 2.0_f32;
        let height = 4.0_f32;

        // Build helix positions and analytic per-node data.
        let mut positions: Vec<[f32; 3]> = Vec::with_capacity(n);
        let mut tangents: Vec<[f32; 3]> = Vec::with_capacity(n);
        let mut normals_3d: Vec<[f32; 3]> = Vec::with_capacity(n);

        for k in 0..n {
            let t = (k as f32 / (n - 1) as f32) * turns * TAU;
            let x = radius * t.cos();
            let y = radius * t.sin();
            let z = height * (k as f32 / (n - 1) as f32) - height * 0.5;
            positions.push([x, y, z]);

            // Unnormalized tangent: d/dt of (r*cos t, r*sin t, h/turns*TAU * t)
            let tx = -radius * t.sin();
            let ty =  radius * t.cos();
            let tz =  height / (turns * TAU);
            let tlen = (tx * tx + ty * ty + tz * tz).sqrt().max(1e-7);
            tangents.push([tx / tlen, ty / tlen, tz / tlen]);

            // Radial outward normal (in XY plane)
            let nx = t.cos();
            let ny = t.sin();
            normals_3d.push([nx, ny, 0.0]);
        }

        // Total segment count for edge quantities.
        let num_segs = n - 1;

        let mut item = PolylineItem::default();
        item.positions = positions.clone();
        item.line_width = self.cnq_line_width;

        match self.cnq_mode {
            CnqMode::EdgeScalar => {
                // Per-edge scalar: arc-length fraction along the helix.
                item.edge_scalars = (0..num_segs)
                    .map(|i| i as f32 / (num_segs - 1) as f32)
                    .collect();
                item.colormap_id = Some(ColormapId(BuiltinColormap::Plasma as usize));
            }

            CnqMode::NodeColor => {
                // Per-node color: RGB rainbow cycling along the strip.
                item.node_colors = (0..n)
                    .map(|k| {
                        let t = k as f32 / (n - 1) as f32;
                        let hue = t * TAU;
                        // Simple HSV->RGB at full saturation and value
                        let h6 = hue * (6.0 / TAU);
                        let hi = h6 as u32 % 6;
                        let f = h6 - hi as f32;
                        let (r, g, b) = match hi {
                            0 => (1.0, f, 0.0),
                            1 => (1.0 - f, 1.0, 0.0),
                            2 => (0.0, 1.0, f),
                            3 => (0.0, 1.0 - f, 1.0),
                            4 => (f, 0.0, 1.0),
                            _ => (1.0, 0.0, 1.0 - f),
                        };
                        [r, g, b, 1.0]
                    })
                    .collect();
            }

            CnqMode::EdgeColor => {
                // Per-edge color: alternating two colors per segment.
                item.edge_colors = (0..num_segs)
                    .map(|i| {
                        if i % 2 == 0 {
                            [0.2, 0.6, 1.0, 1.0]
                        } else {
                            [1.0, 0.4, 0.1, 1.0]
                        }
                    })
                    .collect();
            }

            CnqMode::NodeRadius => {
                // Per-node radius: pulsing sine wave along the strip.
                item.node_radii = (0..n)
                    .map(|k| {
                        let t = k as f32 / (n - 1) as f32;
                        let base = self.cnq_line_width;
                        base * (0.4 + 0.6 * (t * TAU * 3.0).sin().abs())
                    })
                    .collect();
            }

            CnqMode::NodeVectors => {
                // Per-node vectors: scaled tangents.
                item.node_vectors = tangents
                    .iter()
                    .map(|&[tx, ty, tz]| [tx * 0.3, ty * 0.3, tz * 0.3])
                    .collect();
                item.vector_scale = 0.8;
            }

            CnqMode::EdgeVectors => {
                // Per-edge vectors: radial outward normals at each midpoint.
                item.edge_vectors = (0..num_segs)
                    .map(|i| {
                        let n0 = normals_3d[i];
                        let n1 = normals_3d[i + 1];
                        let mx = (n0[0] + n1[0]) * 0.5 * 0.4;
                        let my = (n0[1] + n1[1]) * 0.5 * 0.4;
                        let mz = (n0[2] + n1[2]) * 0.5 * 0.4;
                        [mx, my, mz]
                    })
                    .collect();
                item.vector_scale = 0.8;
            }
        }

        item
    }
}
