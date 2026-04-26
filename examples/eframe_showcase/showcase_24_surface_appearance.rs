//! Showcase 24: Surface Appearance : BackfacePolicy and SSAA.
//!
//! ## BackfacePolicy
//! Three spheres each use a different [`BackfacePolicy`]:
//! - **Left : Cull** (default): back faces are invisible. The clipped interior is dark.
//! - **Centre : Identical**: back faces are shaded the same as front faces.
//! - **Right : DifferentColor**: back faces are shaded red, front faces gray.
//!
//! Spheres use 10×5 tessellation so polygon facets are clearly visible :
//! the same geometry that makes SSAA's effect easy to see on silhouette edges.
//!
//! ## SSAA (Supersampling Anti-Aliasing)
//! A 5×4 grid of small low-poly spheres sits in the background. At 1× the
//! silhouette edges are visibly jagged. At 4× they are smooth. Toggle between
//! factors with the radio buttons and the difference is immediate.
//!
//! This showcase uses [`HdrViewportCallback`](super::hdr_viewport_callback)
//! so SSAA is fully active.

use crate::App;
use eframe::egui;
use glam::Mat4;
use viewport_lib::{
    BackfacePolicy, ClipObject, LightSource, LightingSettings, Material, MeshId,
    PostProcessSettings, SceneRenderItem, ViewportRenderer, scene::Scene,
};

// ---------------------------------------------------------------------------
// App state fields (declared in main App struct):
//   sa_built:       bool
//   sa_scene:       Scene
//   sa_ssaa_factor: u32
//   sa_clip_on:     bool
//   sa_node_ids:    [u64; 3]
// ---------------------------------------------------------------------------

impl App {
    // -------------------------------------------------------------------------
    // One-time scene build
    // -------------------------------------------------------------------------

    pub(crate) fn build_sa_scene(&mut self, renderer: &mut ViewportRenderer) {
        use viewport_lib::geometry::primitives;

        self.sa_scene = Scene::new();

        // Coarse sphere: 10 lon × 5 lat : clearly faceted silhouette edges.
        // This makes aliasing visible at 1× and SSAA's improvement obvious at 4×.
        let sphere_coarse = primitives::sphere(1.2, 10, 5);

        let upload_sphere = |r: &mut ViewportRenderer, device: &eframe::wgpu::Device| -> MeshId {
            MeshId::from_index(
                r.resources_mut()
                    .upload_mesh_data(device, &sphere_coarse)
                    .expect("sa sphere upload"),
            )
        };

        // --- BackfacePolicy demo: three large spheres, clip plane reveals inside ---

        let m0 = upload_sphere(renderer, &self.device);
        let mut mat_cull = Material::from_color([0.7, 0.7, 0.7]);
        mat_cull.backface_policy = BackfacePolicy::Cull;
        let id0 = self.sa_scene.add_named(
            "Cull",
            Some(m0),
            Mat4::from_translation(glam::Vec3::new(-3.0, 0.0, 0.0)),
            mat_cull,
        );

        let m1 = upload_sphere(renderer, &self.device);
        let mut mat_identical = Material::from_color([0.7, 0.7, 0.7]);
        mat_identical.backface_policy = BackfacePolicy::Identical;
        let id1 = self.sa_scene.add_named(
            "Identical",
            Some(m1),
            Mat4::from_translation(glam::Vec3::new(0.0, 0.0, 0.0)),
            mat_identical,
        );

        let m2 = upload_sphere(renderer, &self.device);
        let mut mat_diff = Material::from_color([0.7, 0.7, 0.7]);
        mat_diff.backface_policy = BackfacePolicy::DifferentColor([1.0, 0.1, 0.1]);
        let id2 = self.sa_scene.add_named(
            "DifferentColor",
            Some(m2),
            Mat4::from_translation(glam::Vec3::new(3.0, 0.0, 0.0)),
            mat_diff,
        );

        self.sa_node_ids = [id0, id1, id2];

        // --- SSAA stress-test: 5×4 grid of small coarse spheres in the background ---
        //
        // Small objects approaching pixel-size maximise aliasing at 1× and make
        // the SSAA improvement unmistakable. The grid is placed behind the main
        // spheres so it is always visible regardless of clip-plane state.
        // High tessellation (64×32) so each silhouette edge step is ~1-2px at display size,
        // making aliasing visible at 1× and SSAA's improvement unambiguous at 4×.
        let sphere_tiny = primitives::sphere(0.35, 64, 32);
        let grid_mesh = MeshId::from_index(
            renderer
                .resources_mut()
                .upload_mesh_data(&self.device, &sphere_tiny)
                .expect("sa grid sphere upload"),
        );
        let mut grid_mat = Material::from_color([0.55, 0.65, 0.8]);
        grid_mat.roughness = 0.5;

        for col in 0..5_i32 {
            for row in 0..4_i32 {
                let x = (col - 2) as f32 * 1.6;
                let y = (row - 1) as f32 * 1.4;
                let z = -4.5; // behind the big spheres
                self.sa_scene.add(
                    Some(grid_mesh),
                    Mat4::from_translation(glam::Vec3::new(x, y, z)),
                    grid_mat,
                );
            }
        }

        self.sa_built = true;
    }

    // -------------------------------------------------------------------------
    // Controls panel
    // -------------------------------------------------------------------------

    pub(crate) fn controls_surface_appearance(&mut self, ui: &mut egui::Ui) {
        ui.label("BackfacePolicy : front three spheres:");
        ui.indent("bp_desc", |ui| {
            ui.label("Left:   Cull (interior hidden)");
            ui.label("Centre: Identical (lit normally)");
            ui.label("Right:  DifferentColor (red inside)");
        });
        ui.separator();

        ui.checkbox(&mut self.sa_clip_on, "Clip plane (x = 0)");
        ui.label("Slices each sphere to reveal the\ninner back faces.");

        ui.separator();

        ui.label("SSAA : blue grid in the background:");
        ui.horizontal(|ui| {
            if ui.radio(self.sa_ssaa_factor == 1, "Off (1×)").clicked() {
                self.sa_ssaa_factor = 1;
            }
            if ui.radio(self.sa_ssaa_factor == 2, "2×").clicked() {
                self.sa_ssaa_factor = 2;
            }
            if ui.radio(self.sa_ssaa_factor == 4, "4×").clicked() {
                self.sa_ssaa_factor = 4;
            }
        });
        ui.label("Watch the silhouette edges of the\nsmall spheres. At 1× they are\njagged; at 4× they are smooth.");
    }

    // -------------------------------------------------------------------------
    // Per-frame helpers (called from build_frame_data)
    // -------------------------------------------------------------------------

    pub(crate) fn sa_scene_items(&mut self) -> Vec<SceneRenderItem> {
        self.sa_scene
            .collect_render_items(&viewport_lib::Selection::new())
    }

    pub(crate) fn sa_clip_objects(&self) -> Vec<ClipObject> {
        if self.sa_clip_on {
            vec![ClipObject::plane([-1.0, 0.0, 0.0], 0.0)]
        } else {
            vec![]
        }
    }

    pub(crate) fn sa_post_process(&self) -> PostProcessSettings {
        PostProcessSettings {
            enabled: true,
            ssaa_factor: self.sa_ssaa_factor,
            ..PostProcessSettings::default()
        }
    }

    pub(crate) fn sa_lighting() -> LightingSettings {
        LightingSettings {
            lights: vec![
                LightSource::default(),
                viewport_lib::LightSource {
                    kind: viewport_lib::LightKind::Directional {
                        direction: [-0.5, -0.3, 0.8],
                    },
                    intensity: 0.6,
                    ..viewport_lib::LightSource::default()
                },
            ],
            hemisphere_intensity: 0.3,
            sky_color: [1.0, 1.0, 1.0],
            ground_color: [0.4, 0.4, 0.4],
            ..LightingSettings::default()
        }
    }
}
