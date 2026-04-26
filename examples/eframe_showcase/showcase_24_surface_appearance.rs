//! Showcase 24: Surface Appearance : BackfacePolicy.
//!
//! Three spheres each use a different [`BackfacePolicy`].
//! A clip plane (y = 0) slices all three to reveal the interior :
//! - **Left : Cull** (default): back faces are invisible. The clipped interior is hollow.
//! - **Centre : Identical**: back faces are shaded the same as front faces.
//! - **Right : DifferentColor**: back faces are shaded red, front faces gray.

use crate::App;
use eframe::egui;
use glam::Mat4;
use viewport_lib::{
    BackfacePolicy, ClipObject, LightSource, LightingSettings, Material, MeshId,
    SceneRenderItem, ViewportRenderer, scene::Scene,
};

impl App {
    // -------------------------------------------------------------------------
    // One-time scene build
    // -------------------------------------------------------------------------

    pub(crate) fn build_sa_scene(&mut self, renderer: &mut ViewportRenderer) {
        use viewport_lib::geometry::primitives;

        self.sa_scene = Scene::new();

        // --- Large spheres : one per BackfacePolicy ---
        let sphere_main = primitives::sphere(1.2, 32, 16);
        let upload_main = |r: &mut ViewportRenderer, device: &eframe::wgpu::Device| -> MeshId {
            MeshId::from_index(
                r.resources_mut()
                    .upload_mesh_data(device, &sphere_main)
                    .expect("sa main sphere upload"),
            )
        };

        // Left : Cull
        let m0 = upload_main(renderer, &self.device);
        let mut mat_cull = Material::from_color([0.7, 0.7, 0.7]);
        mat_cull.backface_policy = BackfacePolicy::Cull;
        let id0 = self.sa_scene.add_named(
            "Cull",
            Some(m0),
            Mat4::from_translation(glam::Vec3::new(-3.0, 0.0, 0.0)),
            mat_cull,
        );

        // Centre : Identical
        let m1 = upload_main(renderer, &self.device);
        let mut mat_identical = Material::from_color([0.7, 0.7, 0.7]);
        mat_identical.backface_policy = BackfacePolicy::Identical;
        let id1 = self.sa_scene.add_named(
            "Identical",
            Some(m1),
            Mat4::from_translation(glam::Vec3::new(0.0, 0.0, 0.0)),
            mat_identical,
        );

        // Right : DifferentColor
        let m2 = upload_main(renderer, &self.device);
        let mut mat_diff = Material::from_color([0.7, 0.7, 0.7]);
        mat_diff.backface_policy = BackfacePolicy::DifferentColor([1.0, 0.1, 0.1]);
        let id2 = self.sa_scene.add_named(
            "DifferentColor",
            Some(m2),
            Mat4::from_translation(glam::Vec3::new(3.0, 0.0, 0.0)),
            mat_diff,
        );

        self.sa_node_ids = [id0, id1, id2];

        self.sa_built = true;
    }

    // -------------------------------------------------------------------------
    // Controls panel
    // -------------------------------------------------------------------------

    pub(crate) fn controls_surface_appearance(&mut self, ui: &mut egui::Ui) {
        ui.label("BackfacePolicy (clip plane reveals inside):");
        ui.indent("bp_desc", |ui| {
            ui.label("Left:   Cull (interior hollow)");
            ui.label("Centre: Identical (lit normally)");
            ui.label("Right:  DifferentColor (red/orange)");
        });
        ui.separator();

        ui.checkbox(&mut self.sa_clip_on, "Clip plane (y = 0)");
        ui.label("Slices all spheres to reveal\nhow each policy treats back faces.");
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
            // Clip along Y so the plane slices through all three columns equally,
            // revealing the interior of every sphere.
            vec![ClipObject::plane([0.0, 1.0, 0.0], 0.0)]
        } else {
            vec![]
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
