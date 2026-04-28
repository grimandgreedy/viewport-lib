//! Showcase 17: Volume Rendering & Isosurfaces
//!
//! Demonstrates GPU ray-marching of a 3D scalar volume (`VolumeItem`) and
//! marching-cubes isosurface extraction (`extract_isosurface`) from the same
//! field. The field is a 64×64×64 sum of three Gaussian blobs.
//!
//! Controls:
//! - Mode: Volume / Isosurface / Both
//! - Isovalue slider (re-extracts on release)
//! - Colormap selector for the volume transfer function
//! - Opacity scale slider
//! - Threshold min/max sliders
//! - Step scale slider (quality vs. speed)
//! - Gradient shading toggle
//! - NaN color toggle (render NaN voxels in a distinct colour)
//! - Isosurface material colour and roughness/metallic sliders

use crate::App;
use eframe::egui;
use viewport_lib::{BuiltinColormap, ColormapId, VolumeData, VolumeItem, extract_isosurface};

// ---------------------------------------------------------------------------
// Volume mode enum
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum VolumeMode {
    VolumeOnly,
    IsosurfaceOnly,
    Both,
}

// ---------------------------------------------------------------------------
// App impl
// ---------------------------------------------------------------------------

impl App {
    /// One-time GPU setup for Showcase 17.
    ///
    /// Generates a 64³ scalar field (sum of Gaussian blobs), uploads it to the
    /// GPU as a volume texture, and runs an initial isosurface extraction.
    pub(crate) fn build_volume_scene(&mut self, renderer: &mut viewport_lib::ViewportRenderer) {
        let field = make_gaussian_field(64);

        // Upload the 3D texture.
        let vol_id = renderer.resources_mut().upload_volume(
            &self.device,
            &self.queue,
            &field.data,
            field.dims,
        );
        self.vol_volume_id = Some(vol_id);
        self.vol_field = field;

        // Upload an initial isosurface mesh.
        let iso_mesh = extract_isosurface(&self.vol_field, self.vol_isovalue);
        if !iso_mesh.positions.is_empty() {
            let idx = renderer
                .resources_mut()
                .upload_mesh_data(&self.device, &iso_mesh)
                .expect("isosurface mesh upload");
            self.vol_iso_mesh_index = Some(idx);
        }

        self.vol_built = true;
    }

    /// Re-extract and re-upload the isosurface after an isovalue change.
    pub(crate) fn rebuild_isosurface(&mut self, renderer: &mut viewport_lib::ViewportRenderer) {
        let iso_mesh = extract_isosurface(&self.vol_field, self.vol_isovalue);
        if iso_mesh.positions.is_empty() {
            // No surface at this isovalue : clear the index so nothing is drawn.
            self.vol_iso_mesh_index = None;
            return;
        }
        if let Some(idx) = self.vol_iso_mesh_index {
            // Overwrite the existing slot.
            let _ = renderer
                .resources_mut()
                .replace_mesh_data(&self.device, idx, &iso_mesh);
        } else {
            // Allocate a new slot.
            if let Ok(idx) = renderer
                .resources_mut()
                .upload_mesh_data(&self.device, &iso_mesh)
            {
                self.vol_iso_mesh_index = Some(idx);
            }
        }
    }

    // -------------------------------------------------------------------------
    // Controls panel
    // -------------------------------------------------------------------------

    pub(crate) fn controls_volume(&mut self, ui: &mut egui::Ui, frame: &eframe::Frame) {
        ui.label("Render mode:");
        ui.horizontal(|ui| {
            if ui
                .radio(self.vol_mode == VolumeMode::VolumeOnly, "Volume")
                .clicked()
            {
                self.vol_mode = VolumeMode::VolumeOnly;
            }
            if ui
                .radio(self.vol_mode == VolumeMode::IsosurfaceOnly, "Isosurface")
                .clicked()
            {
                self.vol_mode = VolumeMode::IsosurfaceOnly;
            }
            if ui
                .radio(self.vol_mode == VolumeMode::Both, "Both")
                .clicked()
            {
                self.vol_mode = VolumeMode::Both;
            }
        });

        ui.separator();

        // Isovalue : re-extract only on slider release to avoid GPU stalls.
        ui.label("Isovalue:");
        let iso_resp = ui.add(egui::Slider::new(&mut self.vol_isovalue, 0.01..=0.99).step_by(0.01));
        if iso_resp.drag_stopped() || iso_resp.lost_focus() {
            let rs = frame.wgpu_render_state().expect("wgpu required");
            let mut guard = rs.renderer.write();
            if let Some(renderer) = guard
                .callback_resources
                .get_mut::<viewport_lib::ViewportRenderer>()
            {
                self.rebuild_isosurface(renderer);
            }
        }

        // Volume-specific controls
        if self.vol_mode != VolumeMode::IsosurfaceOnly {
            ui.separator();
            ui.label("Color LUT:");
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
                if ui.radio(self.vol_color_lut == preset, label).clicked() {
                    self.vol_color_lut = preset;
                }
            }

            ui.separator();
            ui.label("Opacity scale:");
            ui.add(egui::Slider::new(&mut self.vol_opacity_scale, 0.1..=4.0).step_by(0.1));

            ui.label("Threshold min:");
            ui.add(egui::Slider::new(&mut self.vol_threshold.0, 0.0..=1.0).step_by(0.01));
            ui.label("Threshold max:");
            ui.add(egui::Slider::new(&mut self.vol_threshold.1, 0.0..=1.0).step_by(0.01));

            ui.label("Step scale (lower = higher quality):");
            ui.add(egui::Slider::new(&mut self.vol_step_scale, 0.25..=4.0).step_by(0.25));

            ui.checkbox(&mut self.vol_shading, "Gradient shading");
            ui.checkbox(&mut self.vol_nan_on, "Show NaN voxels");
        }

        // Isosurface material controls
        if self.vol_mode != VolumeMode::VolumeOnly {
            ui.separator();
            ui.label("Isosurface colour:");
            if ui
                .color_edit_button_rgb(&mut self.vol_iso_material.base_color)
                .changed()
            {}
            ui.label("Roughness:");
            ui.add(
                egui::Slider::new(&mut self.vol_iso_material.roughness, 0.0..=1.0).step_by(0.05),
            );
            ui.label("Metallic:");
            ui.add(egui::Slider::new(&mut self.vol_iso_material.metallic, 0.0..=1.0).step_by(0.05));
        }
    }

    // -------------------------------------------------------------------------
    // Frame-data helpers (called from build_frame_data)
    // -------------------------------------------------------------------------

    /// Build a `VolumeItem` from the current control state.
    pub(crate) fn make_volume_item(&self) -> Option<VolumeItem> {
        let vol_id = self.vol_volume_id?;
        let mut item = VolumeItem::default();
        item.volume_id = vol_id;
        item.color_lut = Some(ColormapId(self.vol_color_lut as usize));
        item.opacity_scale = self.vol_opacity_scale;
        item.scalar_range = (0.0, 1.0);
        item.threshold_min = self.vol_threshold.0;
        item.threshold_max = self.vol_threshold.1;
        item.step_scale = self.vol_step_scale;
        item.enable_shading = self.vol_shading;
        item.nan_color = if self.vol_nan_on {
            Some([0.9, 0.1, 0.9, 0.8])
        } else {
            None
        };
        // Centre the volume around the world origin.
        item.bbox_min = [-3.2, -3.2, -3.2];
        item.bbox_max = [3.2, 3.2, 3.2];
        Some(item)
    }

    /// Build a `SceneRenderItem` for the isosurface mesh.
    pub(crate) fn make_iso_surface_item(&self) -> Option<viewport_lib::SceneRenderItem> {
        let mesh_id = self.vol_iso_mesh_index?;
        let mut item = viewport_lib::SceneRenderItem::default();
        item.mesh_id = mesh_id;
        item.material = self.vol_iso_material;
        // Slight transparency in Both mode so the volume is visible through the surface.
        if self.vol_mode == VolumeMode::Both {
            item.material.opacity = 0.55;
        }
        Some(item)
    }
}

// ---------------------------------------------------------------------------
// Procedural field generation
// ---------------------------------------------------------------------------

/// Generate a 64³ scalar field from a sum of three Gaussian blobs.
///
/// Values are in [0, 1]. Three blobs at asymmetric positions make named-view
/// navigation meaningful: they are visible from different angles.
fn make_gaussian_field(n: u32) -> VolumeData {
    let blobs: &[[f32; 4]] = &[
        // [cx, cy, cz, sigma]
        [0.35, 0.45, 0.50, 0.18],
        [0.65, 0.35, 0.60, 0.14],
        [0.50, 0.65, 0.35, 0.16],
    ];

    let n_usize = n as usize;
    let total = n_usize * n_usize * n_usize;
    let mut data = vec![0.0f32; total];

    for iz in 0..n_usize {
        for iy in 0..n_usize {
            for ix in 0..n_usize {
                let px = ix as f32 / (n as f32 - 1.0);
                let py = iy as f32 / (n as f32 - 1.0);
                let pz = iz as f32 / (n as f32 - 1.0);
                let mut v = 0.0f32;
                for &[cx, cy, cz, sigma] in blobs {
                    let dx = px - cx;
                    let dy = py - cy;
                    let dz = pz - cz;
                    let r2 = dx * dx + dy * dy + dz * dz;
                    v += (-r2 / (2.0 * sigma * sigma)).exp();
                }
                data[ix + iy * n_usize + iz * n_usize * n_usize] = v.min(1.0);
            }
        }
    }

    VolumeData {
        data,
        dims: [n, n, n],
        origin: [0.0, 0.0, 0.0],
        spacing: [1.0 / (n as f32 - 1.0); 3],
    }
}
