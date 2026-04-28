//! Showcase 30: Implicit Surface Rendering & Marching Cubes
//!
//! Demonstrates Phase 13 (CPU sphere-marching) alongside marching cubes, using
//! the same three-sphere SDF scene rendered three ways:
//!
//! - **Merged blobs** : sphere-marching with smooth-min (smin), producing a
//!   single organic fused shape. Color is blended between the three spheres.
//! - **Separate spheres** : sphere-marching with plain min(), so the three
//!   spheres stay distinct with sharp junctions.
//! - **Marching cubes** : the same smin SDF sampled onto a 64³ grid and
//!   triangulated. Renders as a normal mesh; shows the tessellation faceting
//!   that sphere-marching avoids.
//!
//! For the two sphere-marching variants, two mesh spheres (blue=near,
//! orange=far) show depth compositing: toggle "Depth composite" to see the
//! implicit surface interact with scene geometry.

use crate::App;
use eframe::egui;
use glam::Vec3;
use viewport_lib::{
    Camera, GpuImplicitItem, GpuImplicitOptions, GpuMarchingCubesJob, ImplicitBlendMode,
    ImplicitPrimitive, LightKind, LightSource, LightingSettings, Material, SceneRenderItem,
    VolumeData, extract_isosurface,
    geometry::implicit::{ImplicitRenderOptions, march_implicit_surface_color},
    primitives,
};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Which rendering approach to use for the three-sphere SDF scene.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum IsSdfVariant {
    /// Sphere-marching with smooth-min: three spheres fuse into one blob.
    Blobs,
    /// Sphere-marching with hard min: three spheres stay distinct.
    SeparateSpheres,
    /// Marching cubes triangulation of the same smin SDF field.
    MarchingCubes,
    /// GPU implicit surface: same three-blob SDF via descriptor-driven ray-march.
    GpuImplicit,
    /// GPU marching cubes: live isovalue scrubbing via compute shaders.
    GpuMarchingCubes,
}

// ---------------------------------------------------------------------------
// SDF shared kernel
// ---------------------------------------------------------------------------

/// Evaluate the three-blob smin SDF at `p`.
fn blob_sdf(p: Vec3) -> f32 {
    let d0 = (p - Vec3::new(-1.8, 0.0, 0.0)).length() - 1.3;
    let d1 = (p - Vec3::new(1.8, 0.0, 0.0)).length() - 1.3;
    let d2 = (p - Vec3::new(0.0, 1.8, 0.0)).length() - 1.3;
    smin(smin(d0, d1, 0.9), d2, 0.9)
}

/// Per-point color for the blob SDF (proximity-weighted blend of three hues).
fn blob_color(p: Vec3) -> [u8; 4] {
    let d0 = (p - Vec3::new(-1.8, 0.0, 0.0)).length() - 1.3;
    let d1 = (p - Vec3::new(1.8, 0.0, 0.0)).length() - 1.3;
    let d2 = (p - Vec3::new(0.0, 1.8, 0.0)).length() - 1.3;

    const C0: [f32; 3] = [0.90, 0.35, 0.25]; // red-orange
    const C1: [f32; 3] = [0.25, 0.55, 1.00]; // blue
    const C2: [f32; 3] = [0.25, 0.85, 0.45]; // green

    // Bias by the smin blend radius so weights are non-zero on the isosurface
    // (at d_i = 0, weight = 0.9 rather than 0, which would produce black).
    let blend = 0.9_f32;
    let w0 = (-d0 + blend).max(0.0);
    let w1 = (-d1 + blend).max(0.0);
    let w2 = (-d2 + blend).max(0.0);
    let total = (w0 + w1 + w2).max(1e-5);

    [
        ((C0[0] * w0 + C1[0] * w1 + C2[0] * w2) / total * 255.0) as u8,
        ((C0[1] * w0 + C1[1] * w1 + C2[1] * w2) / total * 255.0) as u8,
        ((C0[2] * w0 + C1[2] * w1 + C2[2] * w2) / total * 255.0) as u8,
        255,
    ]
}

// ---------------------------------------------------------------------------
// App methods
// ---------------------------------------------------------------------------

impl App {
    /// Build the showcase: upload reference spheres, the CPU MC mesh, and the GPU MC volume.
    pub(crate) fn build_implicit_scene(&mut self, renderer: &mut viewport_lib::ViewportRenderer) {
        // Small sphere mesh used for the near/far depth-compositing reference objects.
        let sphere = primitives::sphere(0.8, 24, 12);
        self.is_mesh_id = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &sphere)
            .expect("implicit showcase sphere mesh");

        // Marching cubes: sample the blob smin SDF on a 64³ grid over [-3.5, 3.5]³.
        let mc_mesh = {
            let n: u32 = 64;
            let origin = [-3.5_f32; 3];
            let spacing = [7.0 / (n - 1) as f32; 3];
            let mut data = Vec::with_capacity((n * n * n) as usize);
            for iz in 0..n {
                for iy in 0..n {
                    for ix in 0..n {
                        let p = Vec3::new(
                            origin[0] + ix as f32 * spacing[0],
                            origin[1] + iy as f32 * spacing[1],
                            origin[2] + iz as f32 * spacing[2],
                        );
                        data.push(blob_sdf(p));
                    }
                }
            }
            let vol = VolumeData { data, dims: [n, n, n], origin, spacing };
            extract_isosurface(&vol, 0.0)
        };

        if !mc_mesh.positions.is_empty() {
            let id = renderer
                .resources_mut()
                .upload_mesh_data(&self.device, &mc_mesh)
                .expect("marching cubes mesh upload");
            self.is_mc_mesh_id = Some(id);
        }

        // GPU marching cubes: upload a 64³ gyroid scalar field for live isovalue scrubbing.
        {
            let n: u32 = 64;
            let origin = [-4.0_f32; 3];
            let step = 8.0 / (n - 1) as f32;
            let spacing = [step; 3];
            let mut data = Vec::with_capacity((n * n * n) as usize);
            for iz in 0..n {
                for iy in 0..n {
                    for ix in 0..n {
                        let x = origin[0] + ix as f32 * step;
                        let y = origin[1] + iy as f32 * step;
                        let z = origin[2] + iz as f32 * step;
                        // Gyroid surface: sin(x)*cos(y) + sin(y)*cos(z) + sin(z)*cos(x).
                        data.push(x.sin() * y.cos() + y.sin() * z.cos() + z.sin() * x.cos());
                    }
                }
            }
            let vol = VolumeData { data, dims: [n, n, n], origin, spacing };
            let id = renderer
                .resources_mut()
                .upload_volume_for_mc(&self.device, &self.queue, &vol);
            self.is_gmc_volume_id = Some(id);
        }

        self.camera = Camera {
            center: glam::Vec3::new(0.0, 0.5, 0.0),
            distance: 9.0,
            orientation: glam::Quat::from_rotation_z(0.5) * glam::Quat::from_rotation_x(0.9),
            znear: 0.2,
            zfar: 40.0,
            ..Camera::default()
        };

        self.is_built = true;
    }

    /// Side-panel controls for Showcase 30.
    pub(crate) fn controls_implicit(&mut self, ui: &mut egui::Ui) {
        ui.label("Rendering approach:");
        ui.radio_value(
            &mut self.is_sdf_variant,
            IsSdfVariant::GpuImplicit,
            "GPU implicit : descriptor-driven, full resolution",
        );
        ui.radio_value(
            &mut self.is_sdf_variant,
            IsSdfVariant::Blobs,
            "CPU sphere-march : smin (merged blobs)",
        );
        ui.radio_value(
            &mut self.is_sdf_variant,
            IsSdfVariant::SeparateSpheres,
            "CPU sphere-march : min (separate spheres)",
        );
        ui.radio_value(
            &mut self.is_sdf_variant,
            IsSdfVariant::MarchingCubes,
            "Marching cubes : same smin field (CPU)",
        );
        ui.radio_value(
            &mut self.is_sdf_variant,
            IsSdfVariant::GpuMarchingCubes,
            "GPU marching cubes : gyroid field, live isovalue",
        );
        ui.separator();

        if self.is_sdf_variant == IsSdfVariant::GpuMarchingCubes {
            ui.label("Isovalue (gyroid field):");
            ui.add(egui::Slider::new(&mut self.is_gmc_isovalue, -1.5_f32..=1.5).text("isovalue"));
            ui.separator();
        }

        let is_march = self.is_sdf_variant != IsSdfVariant::MarchingCubes
            && self.is_sdf_variant != IsSdfVariant::GpuImplicit
            && self.is_sdf_variant != IsSdfVariant::GpuMarchingCubes;
        ui.add_enabled_ui(is_march, |ui| {
            ui.label("Depth compositing (sphere-march only):");
            ui.checkbox(&mut self.is_depth_composite, "Depth-composite against scene");
            ui.separator();
            ui.label("Render resolution divisor:");
            ui.add(
                egui::Slider::new(&mut self.is_resolution_div, 1_u32..=4)
                    .text("1/N  (lower = faster)"),
            );
        });

        ui.separator();
        ui.label("Blue sphere  : in front of the surface.");
        ui.label("Orange sphere: behind the surface.");
        if self.is_sdf_variant == IsSdfVariant::MarchingCubes {
            ui.separator();
            ui.label("Marching cubes produces a real mesh : orbit and pick it like any other object. Notice the faceting versus the smooth sphere-march result.");
        }
    }

    /// Scene items for Showcase 30.
    ///
    /// Always includes the two reference spheres. Adds the marching-cubes mesh
    /// when that variant is active.
    pub(crate) fn implicit_scene_items(&self) -> Vec<SceneRenderItem> {
        if !self.is_built {
            return vec![];
        }

        let mut items = Vec::new();

        // Near (blue): in front of the blob cloud.
        {
            let mut item = SceneRenderItem::default();
            item.mesh_id = self.is_mesh_id;
            item.model =
                glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.0, 4.5)).to_cols_array_2d();
            item.material = {
                let mut m = Material::from_color([0.25, 0.55, 1.0]);
                m.roughness = 0.35;
                m
            };
            items.push(item);
        }

        // Far (orange): behind the blob cloud.
        {
            let mut item = SceneRenderItem::default();
            item.mesh_id = self.is_mesh_id;
            item.model =
                glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.0, -3.5)).to_cols_array_2d();
            item.material = {
                let mut m = Material::from_color([1.0, 0.48, 0.12]);
                m.roughness = 0.35;
                m
            };
            items.push(item);
        }

        // Marching cubes mesh (MarchingCubes variant only).
        if self.is_sdf_variant == IsSdfVariant::MarchingCubes {
            if let Some(mc_id) = self.is_mc_mesh_id {
                let mut item = SceneRenderItem::default();
                item.mesh_id = mc_id;
                item.model = glam::Mat4::IDENTITY.to_cols_array_2d();
                item.material = {
                    let mut m = Material::from_color([0.80, 0.75, 0.70]);
                    m.roughness = 0.5;
                    m
                };
                items.push(item);
            }
        }

        items
    }

    /// Sphere-march the current SDF variant and push the result into `fd`.
    ///
    /// No-ops when `MarchingCubes` or `GpuImplicit` is active.
    /// Called every frame so the image tracks camera movement.
    pub(crate) fn push_implicit_screen_image(
        &self,
        fd: &mut viewport_lib::FrameData,
        viewport_w: u32,
        viewport_h: u32,
    ) {
        if !self.is_built
            || self.is_sdf_variant == IsSdfVariant::MarchingCubes
            || self.is_sdf_variant == IsSdfVariant::GpuImplicit
            || self.is_sdf_variant == IsSdfVariant::GpuMarchingCubes
        {
            return;
        }

        let div = self.is_resolution_div.max(1);
        let w = (viewport_w / div).max(1);
        let h = (viewport_h / div).max(1);

        let opts = ImplicitRenderOptions {
            width: w,
            height: h,
            max_steps: 128,
            step_scale: 0.85,
            hit_threshold: 5e-4,
            max_distance: self.camera.zfar,
            ..Default::default()
        };

        let cam = &self.camera;
        let mut img = match self.is_sdf_variant {
            IsSdfVariant::Blobs => march_implicit_surface_color(cam, &opts, |p| {
                (blob_sdf(p), blob_color(p))
            }),
            IsSdfVariant::SeparateSpheres => march_implicit_surface_color(cam, &opts, |p| {
                let d0 = (p - Vec3::new(-1.8, 0.0, 0.0)).length() - 1.3;
                let d1 = (p - Vec3::new(1.8, 0.0, 0.0)).length() - 1.3;
                let d2 = (p - Vec3::new(0.0, 1.8, 0.0)).length() - 1.3;
                let d = d0.min(d1).min(d2);
                // Each sphere keeps its own flat color.
                let color = if d0 <= d1 && d0 <= d2 {
                    [230u8, 90, 65, 255]  // red-orange
                } else if d1 <= d2 {
                    [65, 140, 255, 255]   // blue
                } else {
                    [65, 218, 115, 255]   // green
                };
                (d, color)
            }),
            IsSdfVariant::MarchingCubes
            | IsSdfVariant::GpuImplicit
            | IsSdfVariant::GpuMarchingCubes => unreachable!(),
        };

        img.scale = div as f32;

        if !self.is_depth_composite {
            img.depth = None;
        }

        fd.scene.screen_images.push(img);
    }

    /// Submit a GPU implicit surface item for the three-blob scene (Phase 16).
    ///
    /// Only active when `is_sdf_variant == GpuImplicit`. The three sphere
    /// primitives match the CPU-path blob SDF so the two paths are visually
    /// identical (modulo shading differences).
    pub(crate) fn push_gpu_implicit(&self, fd: &mut viewport_lib::FrameData) {
        if !self.is_built || self.is_sdf_variant != IsSdfVariant::GpuImplicit {
            return;
        }

        // Centers and radius matching blob_sdf / blob_color.
        const CENTERS: [[f32; 3]; 3] = [[-1.8, 0.0, 0.0], [1.8, 0.0, 0.0], [0.0, 1.8, 0.0]];
        const COLORS: [[f32; 4]; 3] = [
            [0.90, 0.35, 0.25, 1.0], // red-orange
            [0.25, 0.55, 1.00, 1.0], // blue
            [0.25, 0.85, 0.45, 1.0], // green
        ];

        let mut primitives = Vec::with_capacity(3);
        for i in 0..3 {
            let mut prim = ImplicitPrimitive::zeroed();
            prim.kind = 1; // sphere
            prim.blend = 0.9;
            // params[0..4] = (cx, cy, cz, radius)
            prim.params[0] = CENTERS[i][0];
            prim.params[1] = CENTERS[i][1];
            prim.params[2] = CENTERS[i][2];
            prim.params[3] = 1.3;
            prim.color = COLORS[i];
            primitives.push(prim);
        }

        fd.scene.gpu_implicit.push(GpuImplicitItem {
            primitives,
            blend_mode: ImplicitBlendMode::SmoothUnion,
            march_options: GpuImplicitOptions {
                max_steps: 128,
                step_scale: 0.85,
                hit_threshold: 5e-4,
                max_distance: self.camera.zfar,
            },
        });
    }

    /// Submit a GPU marching cubes job for the gyroid field (Phase 17).
    ///
    /// Only active when `is_sdf_variant == GpuMarchingCubes`.
    pub(crate) fn push_gpu_mc_job(&self, fd: &mut viewport_lib::FrameData) {
        if !self.is_built || self.is_sdf_variant != IsSdfVariant::GpuMarchingCubes {
            return;
        }
        let Some(volume_id) = self.is_gmc_volume_id else { return };

        let mut mat = Material::from_color([0.75, 0.80, 0.85]);
        mat.roughness = 0.4;

        fd.scene.gpu_mc_jobs.push(GpuMarchingCubesJob {
            volume_id,
            isovalue: self.is_gmc_isovalue,
            material: mat,
        });
    }

    /// Lighting for Showcase 30.
    pub(crate) fn implicit_lighting() -> LightingSettings {
        LightingSettings {
            lights: vec![
                LightSource {
                    kind: LightKind::Directional {
                        direction: [0.4, 0.7, 0.9],
                    },
                    color: [1.0, 0.97, 0.93],
                    intensity: 1.4,
                },
                LightSource {
                    kind: LightKind::Directional {
                        direction: [-0.3, 0.2, -0.5],
                    },
                    color: [0.5, 0.6, 0.9],
                    intensity: 0.3,
                },
            ],
            hemisphere_intensity: 0.45,
            sky_color: [0.50, 0.60, 0.80],
            ground_color: [0.25, 0.25, 0.35],
            // Higher bias for the marching-cubes variant: smooth vertex normals on
            // faceted geometry cause shadow-terminator artifacts at the default 0.0001.
            shadow_bias: 0.003,
            ..LightingSettings::default()
        }
    }
}

// ---------------------------------------------------------------------------
// SDF helpers
// ---------------------------------------------------------------------------

/// Smooth minimum (inigo quilez polynomial blend).
#[inline]
fn smin(a: f32, b: f32, k: f32) -> f32 {
    let h = (0.5 + 0.5 * (b - a) / k).clamp(0.0, 1.0);
    a * h + b * (1.0 - h) - k * h * (1.0 - h)
}
