//! Showcase 15: Point Clouds & Glyphs
//!
//! Demonstrates `PointCloudItem` and `GlyphItem` : the two main primitive types
//! for particle and vector-field data. No mesh upload or Scene graph is needed;
//! items are submitted directly to `SceneFrame` each frame.
//!
//! **Point cloud sub-mode:** 20 000-point cloud on a noisy sphere shell, colored
//! by radial distance via a selectable colormap.
//!
//! **Vector field sub-mode:** 5×5×5 grid of arrow glyphs representing an outward-
//! diverging vector field, with magnitude-driven scaling and colormap coloring.

use crate::{App, PcSubMode};
use eframe::egui;
use viewport_lib::{
    BuiltinColormap, ColormapId, GlyphItem, GlyphType, LightingSettings, PointCloudItem,
    SceneRenderItem,
};

impl App {
    /// One-time setup for Showcase 15.
    ///
    /// Generates and caches CPU-side data for both sub-modes. No GPU upload is
    /// required : `PointCloudItem` and `GlyphItem` are submitted directly to
    /// `SceneFrame` each frame.
    pub(crate) fn build_pc_scene(&mut self) {
        let (cloud_pos, cloud_scalars) = make_point_cloud(20_000);
        self.pc_cloud_positions = cloud_pos;
        self.pc_cloud_scalars = cloud_scalars;

        let (field_pos, field_vecs) = make_vector_field();
        self.pc_field_positions = field_pos;
        self.pc_field_vectors = field_vecs;

        self.pc_built = true;
    }

    // -------------------------------------------------------------------------
    // Controls panel
    // -------------------------------------------------------------------------

    pub(crate) fn controls_point_clouds(&mut self, ui: &mut egui::Ui) {
        ui.label("Sub-mode:");
        ui.horizontal(|ui| {
            if ui
                .radio(self.pc_sub_mode == PcSubMode::PointCloud, "Point Cloud")
                .clicked()
            {
                self.pc_sub_mode = PcSubMode::PointCloud;
            }
            if ui
                .radio(self.pc_sub_mode == PcSubMode::VectorField, "Vector Field")
                .clicked()
            {
                self.pc_sub_mode = PcSubMode::VectorField;
            }
        });

        ui.separator();
        ui.label("Colormap:");
        for (preset, label) in [
            (BuiltinColormap::Viridis, "Viridis"),
            (BuiltinColormap::Plasma, "Plasma"),
            (BuiltinColormap::Magma, "Magma"),
            (BuiltinColormap::Inferno, "Inferno"),
            (BuiltinColormap::Turbo, "Turbo"),
            (BuiltinColormap::Greyscale, "Greyscale"),
            (BuiltinColormap::Coolwarm, "Coolwarm"),
            (BuiltinColormap::RdBu, "RdBu"),
            (BuiltinColormap::Rainbow, "Rainbow"),
            (BuiltinColormap::Jet, "Jet"),
        ] {
            if ui.radio(self.pc_colormap == preset, label).clicked() {
                self.pc_colormap = preset;
            }
        }

        ui.separator();
        ui.checkbox(&mut self.pc_scalar_range_manual, "Manual scalar range");
        if self.pc_scalar_range_manual {
            ui.horizontal(|ui| {
                ui.label("Min:");
                ui.add(egui::DragValue::new(&mut self.pc_scalar_range.0).speed(0.01));
            });
            ui.horizontal(|ui| {
                ui.label("Max:");
                ui.add(egui::DragValue::new(&mut self.pc_scalar_range.1).speed(0.01));
            });
        }

        ui.separator();
        match self.pc_sub_mode {
            PcSubMode::PointCloud => {
                ui.label("Point size (px):");
                ui.add(egui::Slider::new(&mut self.pc_point_size, 1.0..=16.0).step_by(0.5));
            }
            PcSubMode::VectorField => {
                ui.label("Glyph type:");
                ui.horizontal(|ui| {
                    if ui
                        .radio(self.pc_glyph_type == GlyphType::Arrow, "Arrow")
                        .clicked()
                    {
                        self.pc_glyph_type = GlyphType::Arrow;
                    }
                    if ui
                        .radio(self.pc_glyph_type == GlyphType::Sphere, "Sphere")
                        .clicked()
                    {
                        self.pc_glyph_type = GlyphType::Sphere;
                    }
                    if ui
                        .radio(self.pc_glyph_type == GlyphType::Cube, "Cube")
                        .clicked()
                    {
                        self.pc_glyph_type = GlyphType::Cube;
                    }
                });

                ui.separator();
                ui.label("Glyph scale:");
                ui.add(egui::Slider::new(&mut self.pc_glyph_scale, 0.1..=3.0).step_by(0.05));

                ui.checkbox(&mut self.pc_glyph_magnitude_scale, "Scale by magnitude");
            }
        }
    }

    // -------------------------------------------------------------------------
    // Frame-data helpers (called from build_frame_data)
    // -------------------------------------------------------------------------

    /// Build a `PointCloudItem` from cached data and current control state.
    pub(crate) fn make_pc_point_cloud_item(&self) -> PointCloudItem {
        let colormap_id = Some(ColormapId(self.pc_colormap as usize));
        let scalar_range = if self.pc_scalar_range_manual {
            Some(self.pc_scalar_range)
        } else {
            None
        };
        let mut item = PointCloudItem::default();
        item.positions = self.pc_cloud_positions.clone();
        item.scalars = self.pc_cloud_scalars.clone();
        item.scalar_range = scalar_range;
        item.colormap_id = colormap_id;
        item.point_size = self.pc_point_size;
        item
    }

    /// Build a `GlyphItem` from cached data and current control state.
    pub(crate) fn make_pc_glyph_item(&self) -> GlyphItem {
        let colormap_id = Some(ColormapId(self.pc_colormap as usize));
        let scalar_range = if self.pc_scalar_range_manual {
            Some(self.pc_scalar_range)
        } else {
            None
        };
        let mut item = GlyphItem::default();
        item.positions = self.pc_field_positions.clone();
        item.vectors = self.pc_field_vectors.clone();
        item.scale = self.pc_glyph_scale;
        item.scale_by_magnitude = self.pc_glyph_magnitude_scale;
        item.scalar_range = scalar_range;
        item.colormap_id = colormap_id;
        item.glyph_type = self.pc_glyph_type;
        item
    }

    /// Standard lighting for Showcase 15.
    pub(crate) fn pc_lighting() -> LightingSettings {
        LightingSettings {
            hemisphere_intensity: 0.5,
            sky_color: [1.0, 1.0, 1.0],
            ground_color: [1.0, 1.0, 1.0],
            ..LightingSettings::default()
        }
    }

    /// Surface items for Showcase 15 (none : all geometry is submitted as
    /// point clouds or glyphs directly on `SceneFrame`).
    pub(crate) fn pc_surface_items() -> Vec<SceneRenderItem> {
        vec![]
    }
}

// ---------------------------------------------------------------------------
// Data generation
// ---------------------------------------------------------------------------

/// Generate `count` points distributed on a noisy sphere shell.
///
/// Returns `(positions, scalars)` where each scalar is the radial distance of
/// that point : useful for demonstrating colormap coloring.
fn make_point_cloud(count: usize) -> (Vec<[f32; 3]>, Vec<f32>) {
    use std::f32::consts::TAU;

    let mut positions = Vec::with_capacity(count);
    let mut scalars = Vec::with_capacity(count);

    // Deterministic LCG for reproducibility across frames.
    let mut seed: u64 = 0xDEAD_BEEF_CAFE_BABE;
    let mut rng = move || -> f32 {
        seed = seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (seed >> 33) as f32 / u32::MAX as f32
    };

    for _ in 0..count {
        let theta = rng() * TAU;
        let phi = (rng() * 2.0 - 1.0).acos();
        let r = 3.0 + rng() * 0.8 - 0.4; // noisy shell: radius 3.0 ± 0.4
        let sp = phi.sin();
        positions.push([r * sp * theta.cos(), r * sp * theta.sin(), r * phi.cos()]);
        scalars.push(r);
    }

    (positions, scalars)
}

/// Generate a 5×5×5 grid of outward-diverging vectors.
///
/// Returns `(positions, vectors)` where each vector points away from the origin
/// with length proportional to radial distance : a classic divergence demo.
fn make_vector_field() -> (Vec<[f32; 3]>, Vec<[f32; 3]>) {
    let mut positions = Vec::new();
    let mut vectors = Vec::new();

    for iz in 0..5_i32 {
        for iy in 0..5_i32 {
            for ix in 0..5_i32 {
                let x = (ix - 2) as f32 * 1.5;
                let y = (iy - 2) as f32 * 1.5;
                let z = (iz - 2) as f32 * 1.5;
                let r = (x * x + y * y + z * z).sqrt();
                let (vx, vy, vz) = if r < 0.01 {
                    (0.0_f32, 0.0, 0.5) // origin: point up
                } else {
                    let scale = 0.4 + r * 0.12;
                    (x / r * scale, y / r * scale, z / r * scale)
                };
                positions.push([x, y, z]);
                vectors.push([vx, vy, vz]);
            }
        }
    }

    (positions, vectors)
}
