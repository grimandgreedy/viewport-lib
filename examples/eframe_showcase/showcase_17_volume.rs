//! Showcase 17: Volume Rendering & Isosurfaces
//!
//! Demonstrates GPU ray-marching of a 3D scalar volume (`VolumeItem`) and
//! marching-cubes isosurface extraction (`extract_isosurface`) from the same
//! field. The field is a 64x64x64 sum of three Gaussian blobs.
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
use viewport_lib::{
    BuiltinColormap, ColormapId, ImageSliceItem, Material, MeshData, MeshId, SliceAxis, VolumeData,
    VolumeId, VolumeItem, VolumeSurfaceSliceItem, extract_isosurface,
};

// ---------------------------------------------------------------------------
// Enum
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum VolumeMode {
    VolumeOnly,
    IsosurfaceOnly,
    Both,
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

pub(crate) struct VolumeState {
    pub built:          bool,
    pub volume_id:      Option<VolumeId>,
    pub iso_mesh_index: Option<MeshId>,
    /// CPU-side field kept for re-extraction on isovalue change.
    pub field:          VolumeData,
    pub mode:           VolumeMode,
    pub isovalue:       f32,
    pub color_lut:      BuiltinColormap,
    pub opacity_scale:  f32,
    pub threshold:      (f32, f32),
    pub step_scale:     f32,
    pub shading:        bool,
    pub nan_on:         bool,
    pub iso_material:   Material,
    /// Whether to overlay an image slice on the volume scene.
    pub show_slice:     bool,
    /// Axis for the image slice (0=X, 1=Y, 2=Z).
    pub slice_axis:     u32,
    /// Normalized [0,1] position of the slice along the axis.
    pub slice_offset:   f32,
    /// Colormap for the image slice LUT.
    pub slice_lut:      BuiltinColormap,
    /// Opacity of the image slice quad.
    pub slice_opacity:  f32,
    /// Whether to overlay a volume surface slice.
    pub show_surface_slice: bool,
    /// Uploaded saddle mesh used as the slice surface.
    pub surface_slice_mesh_id: Option<MeshId>,
    /// Colormap for the surface slice LUT.
    pub surface_slice_lut:     BuiltinColormap,
    /// Opacity of the surface slice.
    pub surface_slice_opacity: f32,
}

impl Default for VolumeState {
    fn default() -> Self {
        let mut iso_material = Material::from_color([0.6, 0.8, 1.0]);
        iso_material.roughness = 0.4;
        Self {
            built:          false,
            volume_id:      None,
            iso_mesh_index: None,
            field: VolumeData {
                data:    Vec::new(),
                dims:    [1, 1, 1],
                origin:  [0.0; 3],
                spacing: [1.0; 3],
            },
            mode:          VolumeMode::VolumeOnly,
            isovalue:      0.35,
            color_lut:     BuiltinColormap::Viridis,
            opacity_scale: 1.0,
            threshold:     (0.05, 1.0),
            step_scale:    1.0,
            shading:       true,
            nan_on:        false,
            iso_material,
            show_slice:    false,
            slice_axis:    2,
            slice_offset:  0.5,
            slice_lut:     BuiltinColormap::Viridis,
            slice_opacity: 1.0,
            show_surface_slice:    false,
            surface_slice_mesh_id: None,
            surface_slice_lut:     BuiltinColormap::Turbo,
            surface_slice_opacity: 1.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Build
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
        self.vol_state.volume_id = Some(vol_id);
        self.vol_state.field = field;

        // Upload an initial isosurface mesh.
        let iso_mesh = extract_isosurface(&self.vol_state.field, self.vol_state.isovalue);
        if !iso_mesh.positions.is_empty() {
            let idx = renderer
                .resources_mut()
                .upload_mesh_data(&self.device, &iso_mesh)
                .expect("isosurface mesh upload");
            self.vol_state.iso_mesh_index = Some(idx);
        }

        self.vol_state.built = true;
    }

    /// Re-extract and re-upload the isosurface after an isovalue change.
    pub(crate) fn rebuild_isosurface(&mut self, renderer: &mut viewport_lib::ViewportRenderer) {
        let iso_mesh = extract_isosurface(&self.vol_state.field, self.vol_state.isovalue);
        if iso_mesh.positions.is_empty() {
            // No surface at this isovalue : clear the index so nothing is drawn.
            self.vol_state.iso_mesh_index = None;
            return;
        }
        if let Some(idx) = self.vol_state.iso_mesh_index {
            // Overwrite the existing slot.
            let _ = renderer
                .resources_mut()
                .replace_mesh_data(&self.device, &self.queue, idx, &iso_mesh);
        } else {
            // Allocate a new slot.
            if let Ok(idx) = renderer
                .resources_mut()
                .upload_mesh_data(&self.device, &iso_mesh)
            {
                self.vol_state.iso_mesh_index = Some(idx);
            }
        }
    }

    // -------------------------------------------------------------------------
    // Frame-data helpers (called from build_frame_data)
    // -------------------------------------------------------------------------

    /// Build a `VolumeItem` from the current control state.
    pub(crate) fn make_volume_item(&self) -> Option<VolumeItem> {
        let s = &self.vol_state;
        let vol_id = s.volume_id?;
        let mut item = VolumeItem::default();
        item.volume_id = vol_id;
        item.color_lut = Some(ColormapId(s.color_lut as usize));
        item.opacity_scale = s.opacity_scale;
        item.scalar_range = (0.0, 1.0);
        item.threshold_min = s.threshold.0;
        item.threshold_max = s.threshold.1;
        item.step_scale = s.step_scale;
        item.enable_shading = s.shading;
        item.nan_color = if s.nan_on { Some([0.9, 0.1, 0.9, 0.8]) } else { None };
        // Centre the volume around the world origin.
        item.bbox_min = [-3.2, -3.2, -3.2];
        item.bbox_max = [3.2, 3.2, 3.2];
        Some(item)
    }

    /// Build an `ImageSliceItem` from the current slice control state.
    pub(crate) fn make_image_slice_item(&self) -> Option<ImageSliceItem> {
        let s = &self.vol_state;
        let vol_id = s.volume_id?;
        let axis = match s.slice_axis {
            0 => SliceAxis::X,
            1 => SliceAxis::Y,
            _ => SliceAxis::Z,
        };
        let mut item = ImageSliceItem::default();
        item.volume_id = vol_id;
        item.axis = axis;
        item.offset = s.slice_offset;
        item.bbox_min = [-3.2, -3.2, -3.2];
        item.bbox_max = [3.2, 3.2, 3.2];
        item.scalar_range = (0.0, 1.0);
        item.color_lut = Some(ColormapId(s.slice_lut as usize));
        item.opacity = s.slice_opacity;
        Some(item)
    }

    /// Upload a saddle-shaped mesh for the surface slice demo if not already done.
    pub(crate) fn ensure_surface_slice_mesh(&mut self, renderer: &mut viewport_lib::ViewportRenderer) {
        if self.vol_state.surface_slice_mesh_id.is_some() {
            return;
        }
        let mesh = make_saddle_mesh(32);
        if let Ok(id) = renderer.resources_mut().upload_mesh_data(&self.device, &mesh) {
            self.vol_state.surface_slice_mesh_id = Some(id);
        }
    }

    /// Build a `VolumeSurfaceSliceItem` from the current surface slice state.
    pub(crate) fn make_volume_surface_slice_item(&self) -> Option<VolumeSurfaceSliceItem> {
        let s = &self.vol_state;
        let vol_id = s.volume_id?;
        let mesh_id = s.surface_slice_mesh_id?;
        let mut item = VolumeSurfaceSliceItem::default();
        item.volume_id = vol_id;
        item.mesh_id = mesh_id;
        item.bbox_min = [-3.2, -3.2, -3.2];
        item.bbox_max = [3.2, 3.2, 3.2];
        item.scalar_range = (0.0, 1.0);
        item.color_lut = Some(ColormapId(s.surface_slice_lut as usize));
        item.opacity = s.surface_slice_opacity;
        Some(item)
    }

    /// Build a `SceneRenderItem` for the isosurface mesh.
    pub(crate) fn make_iso_surface_item(&self) -> Option<viewport_lib::SceneRenderItem> {
        let s = &self.vol_state;
        let mesh_id = s.iso_mesh_index?;
        let mut item = viewport_lib::SceneRenderItem::default();
        item.mesh_id = mesh_id;
        item.material = s.iso_material;
        // Slight transparency in Both mode so the volume is visible through the surface.
        if s.mode == VolumeMode::Both {
            item.material.opacity = 0.55;
        }
        Some(item)
    }
}

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------

pub(crate) fn controls_volume(app: &mut App, ui: &mut egui::Ui, frame: &eframe::Frame) {
    let s = &mut app.vol_state;

    ui.label("Render mode:");
    ui.horizontal(|ui| {
        if ui.radio(s.mode == VolumeMode::VolumeOnly, "Volume").clicked() {
            s.mode = VolumeMode::VolumeOnly;
        }
        if ui.radio(s.mode == VolumeMode::IsosurfaceOnly, "Isosurface").clicked() {
            s.mode = VolumeMode::IsosurfaceOnly;
        }
        if ui.radio(s.mode == VolumeMode::Both, "Both").clicked() {
            s.mode = VolumeMode::Both;
        }
    });

    ui.separator();

    // Isovalue : re-extract only on slider release to avoid GPU stalls.
    ui.label("Isovalue:");
    let iso_resp = ui.add(egui::Slider::new(&mut s.isovalue, 0.01..=0.99).step_by(0.01));
    let need_rebuild = iso_resp.drag_stopped() || iso_resp.lost_focus();

    if need_rebuild {
        let rs = frame.wgpu_render_state().expect("wgpu required");
        let mut guard = rs.renderer.write();
        if let Some(renderer) = guard
            .callback_resources
            .get_mut::<viewport_lib::ViewportRenderer>()
        {
            app.rebuild_isosurface(renderer);
        }
    }

    let s = &mut app.vol_state;

    // Volume-specific controls
    if s.mode != VolumeMode::IsosurfaceOnly {
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
            if ui.radio(s.color_lut == preset, label).clicked() {
                s.color_lut = preset;
            }
        }

        ui.separator();
        ui.label("Opacity scale:");
        ui.add(egui::Slider::new(&mut s.opacity_scale, 0.1..=4.0).step_by(0.1));

        ui.label("Threshold min:");
        ui.add(egui::Slider::new(&mut s.threshold.0, 0.0..=1.0).step_by(0.01));
        ui.label("Threshold max:");
        ui.add(egui::Slider::new(&mut s.threshold.1, 0.0..=1.0).step_by(0.01));

        ui.label("Step scale (lower = higher quality):");
        ui.add(egui::Slider::new(&mut s.step_scale, 0.25..=4.0).step_by(0.25));

        ui.checkbox(&mut s.shading, "Gradient shading");
        ui.checkbox(&mut s.nan_on, "Show NaN voxels");
    }

    // Image slice controls
    ui.separator();
    ui.checkbox(&mut s.show_slice, "Show image slice");
    if s.show_slice {
        ui.label("Slice axis:");
        ui.horizontal(|ui| {
            ui.radio_value(&mut s.slice_axis, 0u32, "X");
            ui.radio_value(&mut s.slice_axis, 1u32, "Y");
            ui.radio_value(&mut s.slice_axis, 2u32, "Z");
        });
        ui.label("Offset:");
        ui.add(egui::Slider::new(&mut s.slice_offset, 0.0..=1.0).step_by(0.01));
        ui.label("Opacity:");
        ui.add(egui::Slider::new(&mut s.slice_opacity, 0.0..=1.0).step_by(0.05));
        ui.label("Color LUT:");
        for (preset, label) in [
            (BuiltinColormap::Viridis, "Viridis"),
            (BuiltinColormap::Turbo, "Turbo"),
            (BuiltinColormap::Greyscale, "Greyscale"),
            (BuiltinColormap::Coolwarm, "Coolwarm"),
        ] {
            if ui.radio(s.slice_lut == preset, label).clicked() {
                s.slice_lut = preset;
            }
        }
    }

    // Isosurface material controls
    if s.mode != VolumeMode::VolumeOnly {
        ui.separator();
        ui.label("Isosurface colour:");
        if ui.color_edit_button_rgb(&mut s.iso_material.base_color).changed() {}
        ui.label("Roughness:");
        ui.add(egui::Slider::new(&mut s.iso_material.roughness, 0.0..=1.0).step_by(0.05));
        ui.label("Metallic:");
        ui.add(egui::Slider::new(&mut s.iso_material.metallic, 0.0..=1.0).step_by(0.05));
    }

    // Volume surface slice controls
    ui.separator();
    ui.checkbox(&mut s.show_surface_slice, "Show surface slice (saddle)");
    if s.show_surface_slice {
        ui.label("Opacity:");
        ui.add(egui::Slider::new(&mut s.surface_slice_opacity, 0.0..=1.0).step_by(0.05));
        ui.label("Color LUT:");
        for (preset, label) in [
            (BuiltinColormap::Turbo, "Turbo"),
            (BuiltinColormap::Viridis, "Viridis"),
            (BuiltinColormap::Coolwarm, "Coolwarm"),
            (BuiltinColormap::Greyscale, "Greyscale"),
        ] {
            if ui.radio(s.surface_slice_lut == preset, label).clicked() {
                s.surface_slice_lut = preset;
            }
        }
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

// ---------------------------------------------------------------------------
// Saddle mesh generation
// ---------------------------------------------------------------------------

/// Generate a saddle surface mesh (z = x^2 - y^2) over [-3.2, 3.2]^2,
/// scaled to sit within the volume bounding box.
fn make_saddle_mesh(n: u32) -> MeshData {
    let range = 3.0f32;
    let scale = 2.5f32; // z amplitude
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    for iy in 0..n {
        for ix in 0..n {
            let fx = ix as f32 / (n - 1) as f32;
            let fy = iy as f32 / (n - 1) as f32;
            let x = (fx * 2.0 - 1.0) * range;
            let y = (fy * 2.0 - 1.0) * range;
            let z = (x * x - y * y) / (range * range) * scale;
            positions.push([x, y, z]);
            // Analytic normal of z = (x^2 - y^2) * k: n = normalize(-dz/dx, -dz/dy, 1)
            let dzdx = 2.0 * x / (range * range) * scale;
            let dzdy = -2.0 * y / (range * range) * scale;
            let len = (dzdx * dzdx + dzdy * dzdy + 1.0f32).sqrt();
            normals.push([-dzdx / len, -dzdy / len, 1.0 / len]);
        }
    }

    for iy in 0..(n - 1) {
        for ix in 0..(n - 1) {
            let base = iy * n + ix;
            indices.push(base);
            indices.push(base + 1);
            indices.push(base + n + 1);
            indices.push(base);
            indices.push(base + n + 1);
            indices.push(base + n);
        }
    }

    let mut mesh = MeshData::default();
    mesh.positions = positions;
    mesh.normals = normals;
    mesh.indices = indices;
    mesh
}
